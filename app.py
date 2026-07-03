"""
app.py — Sales Forecasting & Demand Intelligence Dashboard
4-page Streamlit app for the End-to-End Sales Forecasting Internship Project

Pages:
  1. Sales Overview Dashboard
  2. Forecast Explorer
  3. Anomaly Report
  4. Product Demand Segments

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os

warnings.filterwarnings("ignore")

# ============================================================
#  PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
#  CUSTOM CSS — Premium dark theme
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main { background-color: #0e1117; }

    .metric-card {
        background: linear-gradient(135deg, #1e2130 0%, #252a3d 100%);
        border: 1px solid #2d3452;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 5px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.4);
    }

    .metric-label {
        color: #8b92a5;
        font-size: 0.82rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #e8eaf6;
        font-size: 1.7rem;
        font-weight: 700;
        margin: 0;
    }

    .metric-delta {
        color: #69f0ae;
        font-size: 0.85rem;
        margin-top: 4px;
    }

    .page-title {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }

    .section-header {
        color: #c9d1d9;
        font-size: 1.1rem;
        font-weight: 600;
        border-left: 4px solid #667eea;
        padding-left: 12px;
        margin: 20px 0 10px 0;
    }

    .insight-box {
        background: linear-gradient(135deg, #1a1f35 0%, #1e2540 100%);
        border: 1px solid #3d4f7c;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 10px 0;
        color: #c9d1d9;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    .anomaly-badge {
        background: #ff4b4b20;
        border: 1px solid #ff4b4b;
        border-radius: 20px;
        padding: 4px 12px;
        color: #ff4b4b;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }

    .stSidebar > div:first-child {
        background: linear-gradient(180deg, #141824 0%, #1a2035 100%);
    }

    div[data-testid="stSidebarNav"] { display: none; }

    .stSelectbox label, .stSlider label, .stMultiSelect label {
        color: #8b92a5 !important;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
#  DATA LOADING & CACHING
# ============================================================
@st.cache_data
def load_data():
    """Load and preprocess the Superstore sales dataset."""
    try:
        df = pd.read_csv("train.csv", encoding="latin-1")
    except FileNotFoundError:
        st.error("⚠️ train.csv not found! Please place the Superstore dataset in the same folder as app.py")
        st.stop()

    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    df["Ship Date"]  = pd.to_datetime(df["Ship Date"],  dayfirst=True)
    df["Year"]       = df["Order Date"].dt.year
    df["Month"]      = df["Order Date"].dt.month
    df["Quarter"]    = df["Order Date"].dt.quarter
    df["YearMonth"]  = df["Order Date"].dt.to_period("M")
    df["YearWeek"]   = df["Order Date"].dt.to_period("W")
    df["ShipLag"]    = (df["Ship Date"] - df["Order Date"]).dt.days
    df["Season"]     = df["Month"].apply(
        lambda m: "Winter" if m in [12,1,2] else ("Spring" if m in [3,4,5] else ("Summer" if m in [6,7,8] else "Autumn"))
    )
    return df


@st.cache_data
def compute_monthly_sales(df):
    ms = df.groupby("YearMonth")["Sales"].sum().reset_index()
    ms["YearMonth"] = ms["YearMonth"].dt.to_timestamp()
    ms.columns = ["ds", "y"]
    return ms.sort_values("ds").reset_index(drop=True)


@st.cache_data
def compute_weekly_sales(df):
    ws = df.groupby("YearWeek")["Sales"].sum().reset_index()
    ws["YearWeek"] = ws["YearWeek"].dt.start_time
    ws.columns = ["ds", "y"]
    return ws.sort_values("ds").reset_index(drop=True)


@st.cache_data
def run_anomaly_detection(weekly_df):
    """Run Isolation Forest and Z-Score anomaly detection on weekly data."""
    from sklearn.ensemble import IsolationForest
    ws = weekly_df.copy()
    
    # Isolation Forest
    iso = IsolationForest(contamination=0.08, n_estimators=200, random_state=42)
    iso.fit(ws[["y"]])
    ws["IF_anomaly"] = iso.predict(ws[["y"]]) == -1
    ws["IF_score"]   = iso.score_samples(ws[["y"]])

    # Z-Score
    ws["rolling_mean"] = ws["y"].rolling(window=8, center=True, min_periods=4).mean()
    ws["rolling_std"]  = ws["y"].rolling(window=8, center=True, min_periods=4).std()
    ws["z_score"]      = np.abs((ws["y"] - ws["rolling_mean"]) / ws["rolling_std"])
    ws["ZS_anomaly"]   = ws["z_score"] > 2.0
    ws["both_agree"]   = ws["IF_anomaly"] & ws["ZS_anomaly"]
    return ws


@st.cache_data
def run_clustering(df):
    """Cluster sub-categories by demand patterns."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    years = sorted(df["Year"].unique())

    sub_agg = df.groupby(["Sub-Category", "Year"])["Sales"].sum().reset_index()
    sub_agg = sub_agg.pivot(index="Sub-Category", columns="Year", values="Sales").fillna(0)

    features = pd.DataFrame(index=sub_agg.index)
    features["total_sales"]    = df.groupby("Sub-Category")["Sales"].sum()
    features["growth_rate"]    = ((sub_agg[years[-1]] - sub_agg[years[0]]) /
                                   sub_agg[years[0]].replace(0, np.nan)).fillna(0)

    monthly_sub = df.groupby(["Sub-Category", "YearMonth"])["Sales"].sum()
    features["volatility"]     = monthly_sub.groupby("Sub-Category").std()
    features["avg_order_value"]= df.groupby("Sub-Category")["Sales"].mean()
    features["order_count"]    = df.groupby("Sub-Category")["Sales"].count()
    features = features.dropna()

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=4, n_init=20, random_state=42)
    features["Cluster"] = kmeans.fit_predict(X_scaled)

    cluster_analysis = features.groupby("Cluster").mean()
    cluster_labels = {}
    for cid, row in cluster_analysis.iterrows():
        if (row["total_sales"] > cluster_analysis["total_sales"].median() and
                row["volatility"] < cluster_analysis["volatility"].median()):
            cluster_labels[cid] = "🟢 High Volume, Stable Demand"
        elif row["growth_rate"] > cluster_analysis["growth_rate"].median():
            cluster_labels[cid] = "🚀 Growing Demand"
        elif (row["total_sales"] < cluster_analysis["total_sales"].median() and
              row["volatility"] > cluster_analysis["volatility"].median()):
            cluster_labels[cid] = "🔴 Low Volume, High Volatility"
        else:
            cluster_labels[cid] = "📉 Declining / Mature Demand"

    features["Cluster Label"] = features["Cluster"].map(cluster_labels)

    pca   = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    features["PCA1"] = X_pca[:, 0]
    features["PCA2"] = X_pca[:, 1]

    return features, cluster_labels, pca.explained_variance_ratio_


@st.cache_data
def run_prophet_forecast(monthly_df, n_forecast=3):
    """Train Prophet and return forecast for next n_forecast months."""
    try:
        from prophet import Prophet
        train = monthly_df.iloc[:-n_forecast]
        test  = monthly_df.iloc[-n_forecast:]

        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
            changepoint_prior_scale=0.05
        )
        m.fit(train)
        future   = m.make_future_dataframe(periods=n_forecast, freq="MS")
        forecast = m.predict(future)

        from sklearn.metrics import mean_absolute_error, mean_squared_error
        pred = forecast.iloc[-n_forecast:]["yhat"].values
        mae  = mean_absolute_error(test["y"], pred)
        rmse = np.sqrt(mean_squared_error(test["y"], pred))
        return forecast, mae, rmse, train, test
    except Exception as e:
        return None, None, None, None, None


@st.cache_data
def run_segment_forecast(df, segment_col, segment_val, n_forecast=3):
    """Prophet forecast for a specific segment (category or region)."""
    try:
        from prophet import Prophet
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        seg_df = df[df[segment_col] == segment_val].copy()
        seg_df["YM"] = seg_df["Order Date"].dt.to_period("M")
        monthly = seg_df.groupby("YM")["Sales"].sum().reset_index()
        monthly["YM"] = monthly["YM"].dt.to_timestamp()
        monthly.columns = ["ds", "y"]
        monthly = monthly.sort_values("ds").reset_index(drop=True)

        train = monthly.iloc[:-n_forecast]
        test  = monthly.iloc[-n_forecast:]

        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="multiplicative"
        )
        m.fit(train)
        future   = m.make_future_dataframe(periods=n_forecast, freq="MS")
        forecast = m.predict(future)

        pred = forecast.iloc[-n_forecast:]["yhat"].values
        mae  = mean_absolute_error(test["y"], pred)
        rmse = np.sqrt(mean_squared_error(test["y"], pred))
        return monthly, forecast, pred, mae, rmse
    except Exception as e:
        return None, None, None, None, None


# ============================================================
#  SIDEBAR NAVIGATION
# ============================================================
def sidebar_nav():
    st.sidebar.markdown("""
    <div style='text-align:center; padding: 20px 0 10px 0;'>
        <h1 style='color:#c9d1d9; font-size:1.4rem; font-weight:700; margin:0;'>
            📊 Sales Forecasting
        </h1>
        <p style='color:#8b92a5; font-size:0.8rem; margin:4px 0 0 0;'>
            Demand Intelligence System
        </p>
    </div>
    <hr style='border-color:#2d3452; margin:12px 0;'>
    """, unsafe_allow_html=True)

    pages = {
        "📈  Sales Overview":     "overview",
        "🔮  Forecast Explorer":  "forecast",
        "🚨  Anomaly Report":     "anomaly",
        "🎯  Demand Segments":    "segments"
    }

    page = st.sidebar.radio("Navigate to", list(pages.keys()), label_visibility="collapsed")

    st.sidebar.markdown("""
    <hr style='border-color:#2d3452; margin:12px 0;'>
    <div style='color:#8b92a5; font-size:0.75rem; text-align:center; padding-bottom:10px;'>
        Dataset: Superstore Sales (2014–2017)<br>
        Models: SARIMA · Prophet · XGBoost
    </div>
    """, unsafe_allow_html=True)

    return pages[page]


# ============================================================
#  PAGE 1 — SALES OVERVIEW DASHBOARD
# ============================================================
def page_overview(df, monthly_sales):
    st.markdown('<p class="page-title">📈 Sales Overview Dashboard</p>', unsafe_allow_html=True)
    st.markdown("Interactive exploration of 4 years of retail sales data across categories and regions.")

    # ── Filters
    st.markdown('<p class="section-header">🔧 Filters</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_years = st.multiselect("Year", sorted(df["Year"].unique()), default=sorted(df["Year"].unique()))
    with col2:
        selected_cats = st.multiselect("Category", df["Category"].unique(), default=list(df["Category"].unique()))
    with col3:
        selected_regions = st.multiselect("Region", df["Region"].unique(), default=list(df["Region"].unique()))

    # Apply filters
    mask = (
        df["Year"].isin(selected_years) &
        df["Category"].isin(selected_cats) &
        df["Region"].isin(selected_regions)
    )
    filtered = df[mask]

    if filtered.empty:
        st.warning("No data matches the selected filters.")
        return

    # ── KPI Cards
    st.markdown('<p class="section-header">📊 Key Metrics</p>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    total_rev   = filtered["Sales"].sum()
    total_orders = filtered.shape[0]
    avg_order   = filtered["Sales"].mean()
    total_profit = filtered["Profit"].sum() if "Profit" in filtered.columns else 0

    k1.metric("💰 Total Revenue",  f"${total_rev:,.0f}")
    k2.metric("📦 Total Orders",   f"{total_orders:,}")
    k3.metric("🛒 Avg Order Value", f"${avg_order:,.2f}")
    k4.metric("💵 Total Profit",   f"${total_profit:,.0f}")

    # ── Row 1: Annual bar + Category pie
    st.markdown('<p class="section-header">Revenue Breakdown</p>', unsafe_allow_html=True)
    r1c1, r1c2 = st.columns([3, 2])

    with r1c1:
        yearly = filtered.groupby("Year")["Sales"].sum().reset_index()
        fig_yr = px.bar(
            yearly, x="Year", y="Sales",
            title="Total Sales by Year",
            color="Sales",
            color_continuous_scale="Blues",
            text_auto=".2s"
        )
        fig_yr.update_layout(
            plot_bgcolor="#1e2130", paper_bgcolor="#1e2130",
            font_color="#c9d1d9", showlegend=False,
            title_font_size=14, margin=dict(t=40, b=10, l=10, r=10),
            coloraxis_showscale=False
        )
        fig_yr.update_traces(textposition="outside", textfont_color="#c9d1d9")
        st.plotly_chart(fig_yr, use_container_width=True)

    with r1c2:
        cat_rev = filtered.groupby("Category")["Sales"].sum().reset_index()
        fig_pie = px.pie(
            cat_rev, values="Sales", names="Category",
            title="Revenue by Category",
            hole=0.45,
            color_discrete_sequence=["#4e79a7", "#f28e2b", "#e15759"]
        )
        fig_pie.update_layout(
            plot_bgcolor="#1e2130", paper_bgcolor="#1e2130",
            font_color="#c9d1d9", title_font_size=14,
            margin=dict(t=40, b=10, l=10, r=10)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Row 2: Monthly trend
    st.markdown('<p class="section-header">Monthly Sales Trend</p>', unsafe_allow_html=True)
    monthly_filtered = filtered.groupby("YearMonth")["Sales"].sum().reset_index()
    monthly_filtered["YearMonth"] = monthly_filtered["YearMonth"].dt.to_timestamp()

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=monthly_filtered["YearMonth"], y=monthly_filtered["Sales"],
        mode="lines+markers",
        name="Monthly Sales",
        line=dict(color="#667eea", width=2.5),
        marker=dict(size=5),
        fill="tozeroy",
        fillcolor="rgba(102,126,234,0.08)"
    ))
    rolling_3m = monthly_filtered["Sales"].rolling(3).mean()
    fig_trend.add_trace(go.Scatter(
        x=monthly_filtered["YearMonth"], y=rolling_3m,
        mode="lines", name="3-Month Rolling Mean",
        line=dict(color="#f28e2b", width=2, dash="dash")
    ))
    fig_trend.update_layout(
        title="Monthly Sales Trend (with 3-Month Rolling Average)",
        plot_bgcolor="#1e2130", paper_bgcolor="#1e2130",
        font_color="#c9d1d9", title_font_size=14,
        legend=dict(bgcolor="#252a3d", bordercolor="#3d4f7c"),
        xaxis=dict(showgrid=True, gridcolor="#252a3d"),
        yaxis=dict(showgrid=True, gridcolor="#252a3d", tickprefix="$", tickformat=","),
        margin=dict(t=50, b=30, l=10, r=10)
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # ── Row 3: Region + Sub-category
    r3c1, r3c2 = st.columns(2)
    with r3c1:
        region_yearly = filtered.groupby(["Region", "Year"])["Sales"].sum().reset_index()
        fig_reg = px.line(
            region_yearly, x="Year", y="Sales", color="Region",
            title="Regional Sales Growth by Year",
            markers=True,
            color_discrete_sequence=["#4e79a7","#f28e2b","#e15759","#76b7b2"]
        )
        fig_reg.update_layout(
            plot_bgcolor="#1e2130", paper_bgcolor="#1e2130",
            font_color="#c9d1d9", title_font_size=14,
            legend=dict(bgcolor="#252a3d"),
            xaxis=dict(showgrid=True, gridcolor="#252a3d"),
            yaxis=dict(showgrid=True, gridcolor="#252a3d", tickprefix="$", tickformat=","),
            margin=dict(t=50, b=10, l=10, r=10)
        )
        st.plotly_chart(fig_reg, use_container_width=True)

    with r3c2:
        sub_rev = filtered.groupby("Sub-Category")["Sales"].sum().sort_values(ascending=True).reset_index()
        fig_sub = px.bar(
            sub_rev, x="Sales", y="Sub-Category", orientation="h",
            title="Sales by Sub-Category",
            color="Sales",
            color_continuous_scale="Blues"
        )
        fig_sub.update_layout(
            plot_bgcolor="#1e2130", paper_bgcolor="#1e2130",
            font_color="#c9d1d9", title_font_size=14,
            coloraxis_showscale=False,
            xaxis=dict(showgrid=True, gridcolor="#252a3d", tickprefix="$", tickformat=","),
            margin=dict(t=50, b=10, l=10, r=10)
        )
        st.plotly_chart(fig_sub, use_container_width=True)

    # ── Seasonality heatmap
    st.markdown('<p class="section-header">Seasonality Heatmap</p>', unsafe_allow_html=True)
    season_pivot = filtered.groupby(["Month", "Year"])["Sales"].sum().unstack(fill_value=0)
    month_names  = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    season_pivot.index = [month_names[m-1] for m in season_pivot.index]
    fig_heat = px.imshow(
        season_pivot,
        aspect="auto",
        color_continuous_scale="RdYlGn",
        title="Monthly Sales Heatmap — Identify Seasonal Spikes",
        labels=dict(x="Year", y="Month", color="Sales ($)")
    )
    fig_heat.update_layout(
        plot_bgcolor="#1e2130", paper_bgcolor="#1e2130",
        font_color="#c9d1d9", title_font_size=14,
        margin=dict(t=50, b=10, l=10, r=10)
    )
    st.plotly_chart(fig_heat, use_container_width=True)


# ============================================================
#  PAGE 2 — FORECAST EXPLORER
# ============================================================
def page_forecast(df, monthly_sales):
    st.markdown('<p class="page-title">🔮 Forecast Explorer</p>', unsafe_allow_html=True)
    st.markdown("Explore Prophet-based sales forecasts by category or region.")

    col_ctrl1, col_ctrl2 = st.columns([2, 1])
    with col_ctrl1:
        forecast_type = st.selectbox(
            "Forecast For",
            ["Overall (All Categories & Regions)",
             "Category: Furniture", "Category: Technology", "Category: Office Supplies",
             "Region: West", "Region: East", "Region: Central", "Region: South"]
        )
    with col_ctrl2:
        n_months = st.slider("Forecast Horizon (months)", min_value=1, max_value=3, value=3)

    st.markdown("---")

    # Determine segment
    if forecast_type.startswith("Overall"):
        seg_monthly = monthly_sales
        label = "All Sales"
        seg_col = seg_val = None
    elif forecast_type.startswith("Category:"):
        seg_val = forecast_type.replace("Category: ", "")
        seg_col = "Category"
        label = seg_val
    else:
        seg_val = forecast_type.replace("Region: ", "")
        seg_col = "Region"
        label = seg_val

    # Load forecast
    with st.spinner(f"🔮 Running Prophet forecast for **{label}**..."):
        if seg_col is None:
            forecast, mae, rmse, train, test = run_prophet_forecast(monthly_sales, n_months)
            if forecast is not None:
                hist_data = monthly_sales
        else:
            hist_data, forecast, pred_vals, mae, rmse = run_segment_forecast(df, seg_col, seg_val, n_months)
            if hist_data is not None:
                train = hist_data.iloc[:-n_months]
                test  = hist_data.iloc[-n_months:]

    if forecast is None:
        st.error("⚠️ Could not run forecast. Make sure Prophet is installed: `pip install prophet`")
        return

    # ── Forecast Chart
    test_forecast = forecast.iloc[-n_months:]
    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=train["ds"], y=train["y"],
        mode="lines", name="Historical Sales",
        line=dict(color="#667eea", width=2),
        fill="tozeroy", fillcolor="rgba(102,126,234,0.05)"
    ))

    # Actual test (last n months)
    fig.add_trace(go.Scatter(
        x=test["ds"], y=test["y"],
        mode="lines+markers", name="Actual (Test Period)",
        line=dict(color="#69f0ae", width=3),
        marker=dict(size=10, symbol="circle")
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=test_forecast["ds"], y=test_forecast["yhat"],
        mode="lines+markers", name="Prophet Forecast",
        line=dict(color="#ff6b6b", width=3, dash="dash"),
        marker=dict(size=10, symbol="triangle-up")
    ))

    # Confidence interval
    fig.add_trace(go.Scatter(
        x=pd.concat([test_forecast["ds"], test_forecast["ds"][::-1]]),
        y=pd.concat([test_forecast["yhat_upper"], test_forecast["yhat_lower"][::-1]]),
        fill="toself", fillcolor="rgba(255,107,107,0.15)",
        line=dict(color="rgba(255,0,0,0)"),
        name="95% Confidence Interval"
    ))

    fig.update_layout(
        title=f"Prophet Forecast — {label} ({n_months}-Month Horizon)",
        plot_bgcolor="#1e2130", paper_bgcolor="#1e2130",
        font_color="#c9d1d9", title_font_size=15,
        legend=dict(bgcolor="#252a3d", bordercolor="#3d4f7c"),
        xaxis=dict(showgrid=True, gridcolor="#252a3d", title="Date"),
        yaxis=dict(showgrid=True, gridcolor="#252a3d", tickprefix="$", tickformat=",", title="Sales ($)"),
        margin=dict(t=60, b=40, l=40, r=20),
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Model Metrics
    m1, m2 = st.columns(2)
    m1.metric("📏 MAE (Mean Absolute Error)", f"${mae:,.2f}", help="Average dollar error per month")
    m2.metric("📐 RMSE (Root Mean Sq Error)", f"${rmse:,.2f}", help="Penalizes large errors more heavily")

    # ── Forecast table
    st.markdown('<p class="section-header">Forecast Values</p>', unsafe_allow_html=True)
    forecast_table = test_forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    forecast_table.columns = ["Month", "Forecast ($)", "Lower Bound ($)", "Upper Bound ($)"]
    forecast_table["Month"] = forecast_table["Month"].dt.strftime("%B %Y")
    for col in ["Forecast ($)", "Lower Bound ($)", "Upper Bound ($)"]:
        forecast_table[col] = forecast_table[col].map("${:,.2f}".format)
    st.dataframe(forecast_table, use_container_width=True, hide_index=True)

    # ── Insight box
    st.markdown(f"""
    <div class="insight-box">
        <b>📌 Interpretation:</b> Prophet forecast for <b>{label}</b> over the next <b>{n_months} month(s)</b>
        with an MAE of <b>${mae:,.0f}</b>. The shaded region represents the 95% confidence interval —
        actual sales are expected to fall within this band 95% of the time under normal conditions.
        Large confidence bands indicate higher uncertainty in the forecast.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
#  PAGE 3 — ANOMALY REPORT
# ============================================================
def page_anomaly(df):
    st.markdown('<p class="page-title">🚨 Anomaly Report</p>', unsafe_allow_html=True)
    st.markdown("Detect unusual sales weeks using **Isolation Forest** and **Z-Score** methods.")

    weekly_sales = compute_weekly_sales(df)

    with st.spinner("🔍 Running anomaly detection..."):
        ws = run_anomaly_detection(weekly_sales)

    n_if = ws["IF_anomaly"].sum()
    n_zs = ws["ZS_anomaly"].sum()
    n_both = ws["both_agree"].sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("🌲 Isolation Forest Anomalies", n_if, f"{n_if/len(ws)*100:.1f}% of weeks")
    col2.metric("📊 Z-Score Anomalies",           n_zs, f"{n_zs/len(ws)*100:.1f}% of weeks")
    col3.metric("🔴 Both Methods Agree",           n_both, "High confidence")

    # ── Method 1: Isolation Forest
    st.markdown('<p class="section-header">Method 1: Isolation Forest</p>', unsafe_allow_html=True)

    normal = ws[~ws["IF_anomaly"]]
    anom   = ws[ws["IF_anomaly"]]

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=ws["ds"], y=ws["y"],
        mode="lines", name="Weekly Sales",
        line=dict(color="#667eea", width=1.5),
        fill="tozeroy", fillcolor="rgba(102,126,234,0.05)"
    ))
    fig1.add_trace(go.Scatter(
        x=anom["ds"], y=anom["y"],
        mode="markers", name="⚠️ Anomaly (Isolation Forest)",
        marker=dict(color="#ff4b4b", size=12, symbol="star",
                    line=dict(color="white", width=1))
    ))
    fig1.add_trace(go.Scatter(
        x=ws[ws["both_agree"]]["ds"], y=ws[ws["both_agree"]]["y"],
        mode="markers", name="🔴 Both Methods Agree",
        marker=dict(color="purple", size=16, symbol="x",
                    line=dict(color="white", width=2))
    ))
    fig1.update_layout(
        plot_bgcolor="#1e2130", paper_bgcolor="#1e2130", font_color="#c9d1d9",
        title="Isolation Forest — Weekly Sales Anomalies",
        xaxis=dict(showgrid=True, gridcolor="#252a3d"),
        yaxis=dict(showgrid=True, gridcolor="#252a3d", tickprefix="$", tickformat=","),
        legend=dict(bgcolor="#252a3d"), margin=dict(t=50, b=30, l=40, r=20), height=400
    )
    st.plotly_chart(fig1, use_container_width=True)

    # ── Method 2: Z-Score
    st.markdown('<p class="section-header">Method 2: Z-Score (Rolling Mean ± 2σ)</p>', unsafe_allow_html=True)

    anom_zs = ws[ws["ZS_anomaly"]]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=ws["ds"], y=ws["y"],
        mode="lines", name="Weekly Sales",
        line=dict(color="#667eea", width=1.5)
    ))
    fig2.add_trace(go.Scatter(
        x=ws["ds"], y=ws["rolling_mean"],
        mode="lines", name="Rolling Mean (8-week)",
        line=dict(color="#f28e2b", width=2, dash="dash")
    ))
    # Upper band
    upper = ws["rolling_mean"] + 2 * ws["rolling_std"]
    lower = ws["rolling_mean"] - 2 * ws["rolling_std"]
    fig2.add_trace(go.Scatter(
        x=pd.concat([ws["ds"], ws["ds"][::-1]]),
        y=pd.concat([upper, lower[::-1]]),
        fill="toself", fillcolor="rgba(242,142,43,0.1)",
        line=dict(color="rgba(0,0,0,0)"), name="±2σ Band"
    ))
    fig2.add_trace(go.Scatter(
        x=anom_zs["ds"], y=anom_zs["y"],
        mode="markers", name="⚠️ Anomaly (Z-Score)",
        marker=dict(color="#ff4b4b", size=12, symbol="star",
                    line=dict(color="white", width=1))
    ))
    fig2.update_layout(
        plot_bgcolor="#1e2130", paper_bgcolor="#1e2130", font_color="#c9d1d9",
        title="Z-Score Anomaly Detection — Weekly Sales vs Rolling Mean ± 2σ",
        xaxis=dict(showgrid=True, gridcolor="#252a3d"),
        yaxis=dict(showgrid=True, gridcolor="#252a3d", tickprefix="$", tickformat=","),
        legend=dict(bgcolor="#252a3d"), margin=dict(t=50, b=30, l=40, r=20), height=400
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Anomaly Table
    st.markdown('<p class="section-header">📋 Detected Anomaly Dates</p>', unsafe_allow_html=True)
    anom_table = ws[ws["IF_anomaly"] | ws["ZS_anomaly"]].copy()
    anom_table = anom_table.sort_values("y", ascending=False)

    def get_explanation(row):
        m = row["ds"].month
        if m == 11: return "Black Friday / Thanksgiving — peak consumer sales"
        elif m == 12: return "Christmas/Year-end — holiday and corporate purchases"
        elif m == 9:  return "Back-to-school + Q3 budget cycle"
        elif m == 3:  return "End-of-Q1 corporate spending flush"
        elif m == 1:  return "Post-holiday clearance / New Year purchases"
        else: return "Possible promotional event or demand shock"

    anom_table["Likely Cause"] = anom_table.apply(get_explanation, axis=1)
    display_anom = anom_table[["ds", "y", "IF_anomaly", "ZS_anomaly", "both_agree", "Likely Cause"]].copy()
    display_anom.columns = ["Week Start", "Sales ($)", "IF Flag", "Z-Score Flag", "Both Agree", "Likely Cause"]
    display_anom["Week Start"] = display_anom["Week Start"].dt.strftime("%d %b %Y")
    display_anom["Sales ($)"]  = display_anom["Sales ($)"].map("${:,.2f}".format)
    st.dataframe(display_anom, use_container_width=True, hide_index=True)

    # ── Comparison insight
    st.markdown(f"""
    <div class="insight-box">
        <b>🔍 Method Comparison:</b><br>
        • <b>Isolation Forest</b> detected <b>{n_if} anomalous weeks</b>. It uses a tree-based isolation
          approach and is effective at catching structurally unusual observations even if they are not extreme outliers.<br>
        • <b>Z-Score method</b> flagged <b>{n_zs} anomalous weeks</b>. It compares each week's sales
          to the rolling 8-week average and flags deviations beyond 2 standard deviations.<br>
        • Both methods agreed on <b>{n_both} weeks</b> — these are the <b>highest-confidence anomalies</b>
          requiring immediate business investigation.<br>
        • Discrepancies between methods indicate moderate anomalies that may need human review before action.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
#  PAGE 4 — PRODUCT DEMAND SEGMENTS
# ============================================================
def page_segments(df):
    st.markdown('<p class="page-title">🎯 Product Demand Segments</p>', unsafe_allow_html=True)
    st.markdown("K-Means clustering reveals natural demand groupings across sub-categories.")

    with st.spinner("🔄 Running K-Means clustering..."):
        features, cluster_labels, variance_ratio = run_clustering(df)

    # ── PCA Scatter Plot
    st.markdown('<p class="section-header">2D Cluster Visualization (PCA)</p>', unsafe_allow_html=True)

    color_map = {
        "🟢 High Volume, Stable Demand":  "#69f0ae",
        "🚀 Growing Demand":              "#667eea",
        "🔴 Low Volume, High Volatility": "#ff4b4b",
        "📉 Declining / Mature Demand":   "#f28e2b"
    }

    fig_pca = px.scatter(
        features.reset_index(),
        x="PCA1", y="PCA2",
        color="Cluster Label",
        text="Sub-Category",
        size="total_sales",
        title=f"Product Demand Clusters (PCA) — Explained Variance: {sum(variance_ratio)*100:.1f}%",
        color_discrete_map=color_map,
        size_max=50,
        hover_data={"total_sales": ":$,.0f", "growth_rate": ":.2f", "volatility": ":,.0f"}
    )
    fig_pca.update_traces(textposition="top center", textfont_size=10)
    fig_pca.update_layout(
        plot_bgcolor="#1e2130", paper_bgcolor="#1e2130",
        font_color="#c9d1d9", title_font_size=14,
        legend=dict(bgcolor="#252a3d", bordercolor="#3d4f7c"),
        xaxis=dict(showgrid=True, gridcolor="#252a3d",
                   title=f"PC1 ({variance_ratio[0]*100:.1f}% variance)"),
        yaxis=dict(showgrid=True, gridcolor="#252a3d",
                   title=f"PC2 ({variance_ratio[1]*100:.1f}% variance)"),
        margin=dict(t=60, b=40, l=40, r=20),
        height=550
    )
    st.plotly_chart(fig_pca, use_container_width=True)

    # ── Cluster Overview Metrics
    st.markdown('<p class="section-header">Cluster Summary</p>', unsafe_allow_html=True)
    cluster_summary = features.groupby("Cluster Label").agg(
        Num_SubCategories=("total_sales", "count"),
        Total_Sales=("total_sales", "sum"),
        Avg_Growth_Rate=("growth_rate", "mean"),
        Avg_Volatility=("volatility", "mean"),
        Avg_Order_Value=("avg_order_value", "mean")
    ).reset_index()

    cluster_summary["Total_Sales"]      = cluster_summary["Total_Sales"].map("${:,.0f}".format)
    cluster_summary["Avg_Growth_Rate"]  = cluster_summary["Avg_Growth_Rate"].map("{:.1%}".format)
    cluster_summary["Avg_Volatility"]   = cluster_summary["Avg_Volatility"].map("${:,.0f}".format)
    cluster_summary["Avg_Order_Value"]  = cluster_summary["Avg_Order_Value"].map("${:,.2f}".format)
    cluster_summary.columns = ["Demand Cluster", "# Sub-Categories",
                                "Total Sales", "Avg YoY Growth", "Avg Volatility", "Avg Order Value"]
    st.dataframe(cluster_summary, use_container_width=True, hide_index=True)

    # ── Sub-category assignments
    st.markdown('<p class="section-header">Sub-Category Cluster Assignments</p>', unsafe_allow_html=True)
    sub_table = features.reset_index()[
        ["Sub-Category", "Cluster Label", "total_sales", "growth_rate", "volatility"]
    ].copy()
    sub_table = sub_table.sort_values(["Cluster Label", "total_sales"], ascending=[True, False])
    sub_table["total_sales"] = sub_table["total_sales"].map("${:,.0f}".format)
    sub_table["growth_rate"] = sub_table["growth_rate"].map("{:.1%}".format)
    sub_table["volatility"]  = sub_table["volatility"].map("${:,.0f}".format)
    sub_table.columns = ["Sub-Category", "Demand Cluster", "Total Sales", "YoY Growth", "Sales Volatility"]
    st.dataframe(sub_table, use_container_width=True, hide_index=True)

    # ── Stocking Strategies
    st.markdown('<p class="section-header">📦 Recommended Stocking Strategies</p>', unsafe_allow_html=True)
    strategies = {
        "🟢 High Volume, Stable Demand":  ("✅ Hold high safety stock (30-day buffer). Use Economic Order Quantity (EOQ) model. "
                                           "Negotiate long-term bulk supply contracts to reduce unit cost. "
                                           "Automate replenishment triggers."),
        "🚀 Growing Demand":              ("📈 Increase safety stock buffer by 20-30%. Fast-track new supplier onboarding. "
                                           "Monitor weekly instead of monthly. Set up demand alerts for rapid restocking. "
                                           "Invest in marketing to capture growing market."),
        "🔴 Low Volume, High Volatility": ("⚠️ Keep minimum safety stock. Implement Just-In-Time (JIT) ordering. "
                                           "Consider drop-shipping to avoid holding costs. "
                                           "Review pricing — high volatility may indicate price sensitivity."),
        "📉 Declining / Mature Demand":   ("📉 Gradually reduce inventory levels. Run clearance promotions before seasonal low. "
                                           "Reallocate shelf space & warehouse capacity to growing segments. "
                                           "Evaluate product refresh or discontinuation.")
    }

    for cluster, strategy in strategies.items():
        if cluster in features["Cluster Label"].values:
            subs_in = features[features["Cluster Label"] == cluster].index.tolist()
            st.markdown(f"""
            <div class="insight-box">
                <b>{cluster}</b><br>
                <i>Sub-categories: {", ".join(subs_in)}</i><br><br>
                {strategy}
            </div>
            """, unsafe_allow_html=True)


# ============================================================
#  MAIN APP
# ============================================================
def main():
    df = load_data()
    monthly_sales = compute_monthly_sales(df)

    page = sidebar_nav()

    if page == "overview":
        page_overview(df, monthly_sales)
    elif page == "forecast":
        page_forecast(df, monthly_sales)
    elif page == "anomaly":
        page_anomaly(df)
    elif page == "segments":
        page_segments(df)


if __name__ == "__main__":
    main()
