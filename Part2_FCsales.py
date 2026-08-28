import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.api import Holt, ExponentialSmoothing

# Page Config
st.set_page_config(page_title="Sales & Forecasting Dashboard", layout="wide")

st.title("📊 Sales Performance & Forecasting Dashboard")

# -----------------------------------------------------------------------------
# Data Loading & Preprocessing
# -----------------------------------------------------------------------------
@st.cache_data
def load_data(file_path):
    # Reads Excel file and normalizes column names
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip().str.lower()
    
    # Standardize expected columns
    date_col = [c for c in df.columns if 'date' in c or 'shipped' in c][0]
    channel_col = [c for c in df.columns if 'channel' in c][0]
    item_col = [c for c in df.columns if 'item' in c or 'sku' in c][0]
    qty_col = [c for c in df.columns if 'quantity' in c or 'qty' in c][0]
    rev_col = [c for c in df.columns if 'gross revenue' in c or 'revenue' in c][0]
    profit_col = [c for c in df.columns if 'gross profit' in c or 'profit' in c][0]
    
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Rename for uniform usage
    df = df.rename(columns={
        date_col: 'shipped_date',
        channel_col: 'channel',
        item_col: 'sku_item',
        qty_col: 'quantity',
        rev_col: 'gross_revenue',
        profit_col: 'gross_profit'
    })
    return df

# File loading fallback
try:
    df = load_data('Book2.xlsx')
except Exception:
    uploaded_file = st.sidebar.file_uploader("Upload Sales Data (Book2.xlsx)", type=["xlsx"])
    if uploaded_file:
        df = load_data(uploaded_file)
    else:
        st.warning("Please place 'Book2.xlsx' in the working directory or upload it via sidebar.")
        st.stop()

# -----------------------------------------------------------------------------
# SECTION 1: CHANNEL PERFORMANCE
# -----------------------------------------------------------------------------
st.header("1. Channel Performance Analysis")

# Filters
col1, col2 = st.columns(2)

all_channels = sorted(df['channel'].dropna().unique().tolist())
metric_map = {
    'Quantity': 'quantity',
    'Gross Revenue': 'gross_revenue',
    'Gross Profit': 'gross_profit'
}

with col1:
    selected_channels = st.multiselect(
        "Select Channel(s):",
        options=all_channels,
        default=all_channels
    )

with col2:
    selected_metrics = st.multiselect(
        "Select Metric(s):",
        options=list(metric_map.keys()),
        default=['Quantity']
    )

if not selected_channels or not selected_metrics:
    st.warning("Please select at least one channel and one metric.")
else:
    # Aggregation by Month and Channel
    df_channel = df[df['channel'].isin(selected_channels)].copy()
    df_channel['month'] = df_channel['shipped_date'].dt.to_period('M').dt.to_timestamp()
    
    selected_cols = [metric_map[m] for m in selected_metrics]
    channel_grouped = df_channel.groupby(['month', 'channel'])[selected_cols].sum().reset_index()

    # Tabs for Section 1
    tab1_ch, tab2_ch = st.tabs(["📈 Line Chart View", "📋 Data Table View"])

    with tab1_ch:
        fig = go.Figure()
        for metric in selected_metrics:
            col_name = metric_map[metric]
            for ch in selected_channels:
                ch_data = channel_grouped[channel_grouped['channel'] == ch]
                fig.add_trace(go.Scatter(
                    x=ch_data['month'],
                    y=ch_data[col_name],
                    mode='lines+markers',
                    name=f"{ch} - {metric}"
                ))
        fig.update_layout(
            title="Channel Performance Over Time",
            xaxis_title="Shipped Date (Month)",
            yaxis_title="Value",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2_ch:
        st.dataframe(channel_grouped, use_container_width=True)

st.divider()

# -----------------------------------------------------------------------------
# SECTION 2: SKU FORECASTING (2022 NEXT 6 MONTHS)
# -----------------------------------------------------------------------------
st.header("2. SKU Item Quantity Forecast (Jan - Jun 2022)")

all_items = sorted(df['sku_item'].dropna().unique().tolist())
selected_item = st.selectbox("Select SKU Item:", options=all_items)

# Prepare monthly quantity time series
df_item = df[df['sku_item'] == selected_item].copy()
df_item['month'] = df_item['shipped_date'].dt.to_period('M').dt.to_timestamp()
ts = df_item.groupby('month')['quantity'].sum().asfreq('MS', fill_value=0)

# Filter history up to Dec 2021
ts_2021 = ts[ts.index.year == 2021]

if len(ts_2021) < 3:
    # If 2021 subset is empty, fallback to available historical series
    ts_hist = ts
else:
    ts_hist = ts_2021

# --- Part 1: Pattern Recognition Logic ---
def classify_pattern(series):
    if len(series) < 4:
        return "Others", "3-Month Moving Average"
    
    vals = series.values
    x = np.arange(len(vals))
    
    # Calculate linear correlation coefficient
    if np.std(vals) > 0:
        corr = np.corrcoef(x, vals)[0, 1]
    else:
        corr = 0
        
    # Check simple seasonal variability (peak-to-trough cadence or autocorrelation)
    diff = np.diff(vals)
    sign_switches = np.sum(np.diff(np.sign(diff)) != 0) if len(diff) > 1 else 0
    
    # Rules
    if sign_switches >= (len(vals) // 3) and np.std(vals) / (np.mean(vals) + 1e-5) > 0.3:
        pattern = "Seasonality"
        model_name = "Winters' Exponential Smoothing"
    elif abs(corr) >= 0.45:
        trend_dir = "Positive" if corr > 0 else "Negative"
        pattern = f"Trend ({trend_dir})"
        model_name = "Holt's Exponential Smoothing"
    else:
        pattern = "Others"
        model_name = "3-Month Moving Average"
        
    return pattern, model_name

pattern, model_name = classify_pattern(ts_hist)

# Display Pattern Notification
st.info(f"**Data Pattern Detected:** {pattern}  \n**Assigned Model:** {model_name}")

# --- Part 2: Forecasting Execution ---
future_dates = pd.date_range(start="2022-01-01", periods=6, freq="MS")

if "Holt's" in model_name:
    try:
        model = Holt(ts_hist, initialization_method="estimated").fit()
        forecast_vals = model.forecast(6)
    except Exception:
        forecast_vals = pd.Series([ts_hist.tail(3).mean()] * 6, index=future_dates)

elif "Winters'" in model_name:
    try:
        # Fallback period logic if series length < 12
        periods = 3 if len(ts_hist) < 12 else 12
        model = ExponentialSmoothing(
            ts_hist, 
            trend="add", 
            seasonal="add", 
            seasonal_periods=periods,
            initialization_method="estimated"
        ).fit()
        forecast_vals = model.forecast(6)
    except Exception:
        # Robust fallback to Holt or moving average if seasonal fit fails
        forecast_vals = pd.Series([ts_hist.tail(3).mean()] * 6, index=future_dates)

else:  # 3-Month Moving Average
    ma_val = ts_hist.tail(3).mean()
    forecast_vals = pd.Series([ma_val] * 6, index=future_dates)

# Clip negative values to zero
forecast_vals = forecast_vals.clip(lower=0)

# Build Display Dataframe
df_forecast = pd.DataFrame({
    'Month': future_dates.strftime('%Y-%m'),
    'Forecasted Quantity': np.round(forecast_vals.values, 2)
})

# Display Tabs for Section 2
tab1_fc, tab2_fc = st.tabs(["📈 Visualization", "📋 Forecast Data Table"])

with tab1_fc:
    fig_fc = go.Figure()

    # Historical Line
    fig_fc.add_trace(go.Scatter(
        x=ts_hist.index,
        y=ts_hist.values,
        mode='lines+markers',
        name='Historical Quantity (2021)'
    ))

    # Forecast Line
    fig_fc.add_trace(go.Scatter(
        x=future_dates,
        y=forecast_vals.values,
        mode='lines+markers',
        name='Forecast Quantity (2022 Jan-Jun)',
        line=dict(dash='dash', color='orange')
    ))

    fig_fc.update_layout(
        title=f"2021 Actuals vs 2022 H1 Forecast for {selected_item}",
        xaxis_title="Month",
        yaxis_title="Quantity",
        hovermode="x unified"
    )
    st.plotly_chart(fig_fc, use_container_width=True)

with tab2_fc:
    st.dataframe(df_forecast, use_container_width=True)