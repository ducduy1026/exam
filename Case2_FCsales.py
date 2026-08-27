import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.api import Holt, ExponentialSmoothing

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

st.title("📊 Monthly Sales Forecasting Dashboard (Year 6)")

# File loader
df = pd.read_excel("Book1.xlsx")

# Data preparation
data_actual = df.dropna(subset=['Sales']).copy()
actual_series = data_actual['Sales'].values
dates_hist = data_actual['Date'].tolist()
y6_dates = [f"Y6/{i:02d}" for i in range(1, 13)]
all_dates = dates_hist + y6_dates

# 1. 3-Month Moving Average
ma3_fit = pd.Series(actual_series).rolling(window=3).mean().shift(1).values
ma3_history = list(actual_series[-3:])
ma3_fc = []
for _ in range(12):
    val = np.mean(ma3_history[-3:])
    ma3_fc.append(val)
    ma3_history.append(val)
ma3_full = np.concatenate([ma3_fit, ma3_fc])

# 2. Holt's Exponential Smoothing
holt_model = Holt(actual_series, initialization_method="estimated").fit()
holt_fit = holt_model.fittedvalues
holt_fc = holt_model.forecast(12)
holt_full = np.concatenate([holt_fit, holt_fc])

# 3. Winters' Exponential Smoothing
winters_model = ExponentialSmoothing(
    actual_series, trend='add', seasonal='mul', seasonal_periods=12, initialization_method="estimated"
).fit()
winters_fit = winters_model.fittedvalues
winters_fc = winters_model.forecast(12)
winters_full = np.concatenate([winters_fit, winters_fc])

# Calculate Error Metrics
def calc_metrics(y_true, y_pred):
    mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
    err = y_true[mask] - y_pred[mask]
    return {
        'Forecast Bias': np.mean(err),
        'MAD': np.mean(np.abs(err)),
        'MAPE (%)': np.mean(np.abs(err / y_true[mask])) * 100,
        'MSE': np.mean(err**2)
    }

metrics_dict = {
    '3-Month Moving Average': calc_metrics(actual_series, ma3_fit),
    "Holt's Exponential Smoothing": calc_metrics(actual_series, holt_fit),
    "Winters' Exponential Smoothing": calc_metrics(actual_series, winters_fit)
}
metrics_df = pd.DataFrame(metrics_dict).T.round(2)

st.subheader("Forecast Error Comparison")
st.dataframe(metrics_df, use_container_width=True)

# Multiselect filter in sidebar
st.sidebar.header("🔍 Filter Options")
methods_list = ['3-Month Moving Average', "Holt's Exponential Smoothing", "Winters' Exponential Smoothing"]
selected_methods = st.sidebar.multiselect(
    "Choose Forecasting Method(s):",
    options=methods_list,
    default=methods_list
)

# Build combined master dataset
full_df = pd.DataFrame({'Date': all_dates})
actual_padded = np.append(actual_series, [np.nan] * 12)
full_df['Actual Sales'] = actual_padded
full_df['3-Month Moving Average'] = ma3_full
full_df["Holt's Exponential Smoothing"] = holt_full
full_df["Winters' Exponential Smoothing"] = winters_full

# Tabbed Layout
st.subheader("Forecast Visualization & Data Table")
tab1, tab2 = st.tabs(["📈 Line Chart", "📋 Data Table"])

with tab1:
    fig = go.Figure()
    
    # Historical Actual Line
    fig.add_trace(go.Scatter(
        x=full_df['Date'][:60], 
        y=full_df['Actual Sales'][:60],
        mode='lines+markers',
        name='Actual Sales (Y1-Y5)',
        line=dict(color="#6A6A6D", width=2.5)
    ))
    
    # Selected Forecasting Lines
    color_map = {
        '3-Month Moving Average': '#EF553B',
        "Holt's Exponential Smoothing": '#00CC96',
        "Winters' Exponential Smoothing": '#636EFA'
    }
    
    for method in selected_methods:
        fig.add_trace(go.Scatter(
            x=full_df['Date'],
            y=full_df[method],
            mode='lines',
            name=method,
            line=dict(color=color_map[method], width=2, dash='dash')
        ))
        
    fig.update_layout(
        title="Monthly Sales & Year 6 Forecast Comparison",
        xaxis_title="Date",
        yaxis_title="Sales Units",
        hovermode="x unified",
        template="plotly_white",
        height=550
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    cols_to_display = ['Date', 'Actual Sales'] + selected_methods
    st.dataframe(full_df[cols_to_display].round(2), use_container_width=True)