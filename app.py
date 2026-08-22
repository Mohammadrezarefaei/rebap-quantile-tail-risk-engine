import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from src.risk_metrics import calculate_imbalance_pnl, compute_var_cvar

st.set_page_config(page_title="Quantile reBAP Tail-Risk Engine", layout="wide")

st.title("⚡ Quantile reBAP Forecaster & Imbalance Tail-Risk Engine")
st.markdown(
    "Stress-testing Day-Ahead scheduling error distributions against extreme German **reBAP settlement spikes** using **CVaR**."
)

# Sidebar Parameters
st.sidebar.header("Simulation Parameters")
n_hours = st.sidebar.slider("Test Horizon (Hours)", 48, 720, 168)
wind_ramp_severity = st.sidebar.slider("Wind Ramp Multiplier (Stress Factor)", 1.0, 3.0, 1.5)
confidence_level = st.sidebar.slider("Confidence Level for CVaR", 0.90, 0.99, 0.95)

# Generate Synthetic German Balancing Data
np.random.seed(42)
time_index = pd.date_range("2026-01-01", periods=n_hours, freq="h")

# Base profiles
base_load = 50 + 15 * np.sin(np.linspace(0, 8 * np.pi, n_hours))
wind_generation = 20 + 10 * np.cos(np.linspace(0, 4 * np.pi, n_hours)) * wind_ramp_severity
actual_demand = base_load - wind_generation + np.random.normal(0, 2.5, n_hours)

# Point forecast vs Quantiles
p50_forecast = base_load - wind_generation
p10_forecast = p50_forecast - 4.5
p90_forecast = p50_forecast + 4.5

# Prices (EUR/MWh) with intermittent reBAP spikes
da_price = 75 + 12 * np.sin(np.linspace(0, 6 * np.pi, n_hours)) + np.random.normal(0, 5, n_hours)
rebap_spikes = np.random.choice([0, 1], size=n_hours, p=[0.92, 0.08]) * np.random.uniform(250, 650, n_hours)
rebap_price = da_price + np.random.normal(0, 15, n_hours) + rebap_spikes

# Evaluate PnL and Tail Risk
df_results = calculate_imbalance_pnl(actual_demand, p50_forecast, da_price, rebap_price)
risk_stats = compute_var_cvar(df_results["cash_penalty_eur"].values, confidence_level)

# KPI Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Mean Cash Exposure", f"€{df_results['cash_penalty_eur'].mean():,.2f}")
col2.metric("Total P&L Drawdown", f"€{df_results['cash_penalty_eur'].sum():,.2f}")
col3.metric(f"VaR ({int(confidence_level*100)}%)", f"€{risk_stats['var']:,.2f}")
col4.metric(f"CVaR (Tail Risk)", f"€{risk_stats['cvar']:,.2f}")

# Main Visualizations
st.subheader("Day-Ahead Schedule vs. Probabilistic Quantile Bounds (P10 - P90)")
fig_quantiles = go.Figure()
fig_quantiles.add_trace(go.Scatter(x=time_index, y=p90_forecast, line=dict(width=0), showlegend=False, name="P90"))
fig_quantiles.add_trace(go.Scatter(x=time_index, y=p10_forecast, line=dict(width=0), fill='tonexty', fillcolor='rgba(0,100,255,0.15)', name="P10-P90 Quantile Band"))
fig_quantiles.add_trace(go.Scatter(x=time_index, y=actual_demand, mode="lines", name="Actual Demand (GW)", line=dict(color="black", width=2)))
fig_quantiles.add_trace(go.Scatter(x=time_index, y=p50_forecast, mode="lines", name="P50 Forecast (DA Schedule)", line=dict(color="blue", dash="dash")))
st.plotly_chart(fig_quantiles, use_container_width=True)

st.subheader("reBAP Settlement Spikes vs Financial Drawdowns (EUR)")
fig_tail = go.Figure()
fig_tail.add_trace(go.Bar(x=time_index, y=df_results["cash_penalty_eur"], name="Financial Cash Penalty (EUR)", marker_color="crimson"))
fig_tail.add_trace(go.Scatter(x=time_index, y=rebap_price, mode="lines", name="reBAP Price (EUR/MWh)", yaxis="y2", line=dict(color="orange", width=1.5)))
fig_tail.update_layout(
    yaxis=dict(title="Cash Penalty (€)"),
    yaxis2=dict(title="reBAP Price (€/MWh)", overlaying="y", side="right")
)
st.plotly_chart(fig_tail, use_container_width=True)
