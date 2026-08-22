import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------
# Page Configuration & Custom CSS Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Quantile reBAP Tail-Risk Engine",
    page_icon="⚡",
    layout="wide"
)

# Custom Styling: Crisp Dark UI with Blue Accents
st.markdown("""
<style>
    /* Metric Card Styling with Blue Borders */
    div[data-testid="stMetric"] {
        background-color: #0b1329;
        border: 1.5px solid #2563eb;
        padding: 16px 20px;
        border-radius: 10px;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.15);
    }
    div[data-testid="stMetricLabel"] {
        color: #93c5fd;
        font-size: 0.88rem;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff;
        font-size: 1.6rem;
        font-weight: 700;
    }
    /* Section Divider */
    hr {
        border-top: 1px solid #1e3a8a;
        margin: 25px 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Title & Description
# ---------------------------------------------------------
st.title("⚡ Quantile reBAP Forecaster & Tail-Risk Engine")
st.markdown(
    "<p style='color: #94a3b8; font-size: 1.05rem; margin-top: -10px;'>"
    "Stress-testing Day-Ahead scheduling error distributions against extreme German "
    "<b style='color: #f59e0b;'>reBAP settlement spikes</b> using <b>CVaR (Conditional Value at Risk)</b>."
    "</p>",
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Risk Engine Math
# ---------------------------------------------------------
def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    error = y_true - y_pred
    return float(np.mean(np.maximum(quantile * error, (quantile - 1) * error)))

def calculate_imbalance_pnl(actual_mwh, forecast_mwh, da_price_eur, rebap_price_eur):
    residual_mwh = actual_mwh - forecast_mwh
    spread_eur = rebap_price_eur - da_price_eur
    cash_penalty_eur = -1.0 * residual_mwh * spread_eur
    return pd.DataFrame({
        "actual_mwh": actual_mwh,
        "forecast_mwh": forecast_mwh,
        "residual_mwh": residual_mwh,
        "da_price_eur": da_price_eur,
        "rebap_price_eur": rebap_price_eur,
        "cash_penalty_eur": cash_penalty_eur
    })

def compute_var_cvar(penalties: np.ndarray, confidence_level: float = 0.95):
    clean = penalties[~np.isnan(penalties)]
    losses = clean[clean > 0]
    if len(losses) == 0:
        return {"var": 0.0, "cvar": 0.0}
    var_th = float(np.percentile(losses, confidence_level * 100))
    cvar_val = float(np.mean(losses[losses >= var_th]))
    return {"var": var_th, "cvar": cvar_val}

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.markdown("### ⚙️ Simulation Settings")
n_hours = st.sidebar.slider("Simulation Horizon (Hours)", min_value=48, max_value=720, value=168, step=24)
wind_ramp_severity = st.sidebar.slider("Wind Ramp Stress Factor", min_value=1.0, max_value=3.0, value=1.6, step=0.1)
spike_prob = st.sidebar.slider("reBAP Spike Frequency (%)", min_value=2, max_value=20, value=8) / 100.0
confidence_level = st.sidebar.slider("CVaR Confidence Level", min_value=0.85, max_value=0.99, value=0.95, step=0.01)

# ---------------------------------------------------------
# Synthetic Market Data Generator
# ---------------------------------------------------------
np.random.seed(42)
time_index = pd.date_range("2026-08-01 00:00", periods=n_hours, freq="h")

base_load = 50.0 + 15.0 * np.sin(np.linspace(0, 8 * np.pi, n_hours))
wind_gen = (18.0 + 10.0 * np.cos(np.linspace(0, 4 * np.pi, n_hours))) * wind_ramp_severity
actual_demand = base_load - wind_gen + np.random.normal(0, 1.8, n_hours)

p50_schedule = base_load - wind_gen
uncertainty = 3.2 + 1.2 * np.abs(np.cos(np.linspace(0, 4 * np.pi, n_hours)))
p10_schedule = p50_schedule - uncertainty
p90_schedule = p50_schedule + uncertainty

da_price = 70.0 + 12.0 * np.sin(np.linspace(0, 6 * np.pi, n_hours)) + np.random.normal(0, 3.5, n_hours)
spikes = np.random.choice([0, 1], size=n_hours, p=[1 - spike_prob, spike_prob]) * np.random.uniform(250, 700, n_hours)
rebap_price = da_price + np.random.normal(0, 8.0, n_hours) + spikes

df_sim = calculate_imbalance_pnl(actual_demand * 100, p50_schedule * 100, da_price, rebap_price)
risk_stats = compute_var_cvar(df_sim["cash_penalty_eur"].values, confidence_level)

p10_loss = pinball_loss(actual_demand, p10_schedule, 0.10)
p50_loss = pinball_loss(actual_demand, p50_schedule, 0.50)
p90_loss = pinball_loss(actual_demand, p90_schedule, 0.90)

# ---------------------------------------------------------
# KPI Cards
# ---------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Mean Hourly Exposure", f"€{df_sim['cash_penalty_eur'].mean():,.2f}")
col2.metric("Total P&L Drawdown", f"€{df_sim['cash_penalty_eur'].sum():,.2f}")
col3.metric(f"VaR ({int(confidence_level*100)}%)", f"€{risk_stats['var']:,.2f}")
col4.metric(f"CVaR (Tail Risk @ {int(confidence_level*100)}%)", f"€{risk_stats['cvar']:,.2f}")

st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Reusable Blue-Themed Legend Configuration
# ---------------------------------------------------------
blue_legend_style = dict(
    orientation="h",
    yanchor="bottom",
    y=1.03,
    xanchor="right",
    x=1,
    bgcolor="rgba(11, 24, 52, 0.95)",  # Deep navy background
    bordercolor="#3b82f6",             # Crisp Electric Blue Border
    borderwidth=1.5,
    font=dict(color="#f8fafc", size=11, family="sans-serif")
)

# ---------------------------------------------------------
# Plot 1: Dispatch vs Quantile Bands
# ---------------------------------------------------------
st.markdown("#### 1. Day-Ahead Dispatch vs. Probabilistic Quantile Bounds (P10 - P90)")

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=time_index, y=p90_schedule, line=dict(width=0),
    showlegend=False, name="P90 Upper Bound"
))
fig1.add_trace(go.Scatter(
    x=time_index, y=p10_schedule, line=dict(width=0),
    fill="tonexty", fillcolor="rgba(59, 130, 246, 0.25)",
    name="P10-P90 Quantile Band"
))
fig1.add_trace(go.Scatter(
    x=time_index, y=actual_demand, mode="lines",
    name="Actual Grid Demand (GW)", line=dict(color="#ffffff", width=2.2)
))
fig1.add_trace(go.Scatter(
    x=time_index, y=p50_schedule, mode="lines",
    name="P50 DA Schedule (GW)", line=dict(color="#60a5fa", dash="dash", width=2.0)
))

fig1.update_layout(
    template="plotly_dark",
    plot_bgcolor="#0b0f19",
    paper_bgcolor="#0b0f19",
    height=400,
    margin=dict(l=20, r=20, t=45, b=20),
    legend=blue_legend_style,
    xaxis=dict(gridcolor="#1e293b", title="Timeline"),
    yaxis=dict(gridcolor="#1e293b", title="Power [GW]"),
    hovermode="x unified"
)
st.plotly_chart(fig1, use_container_width=True)

# ---------------------------------------------------------
# Plot 2: Subplots for Cash Penalties & reBAP Spikes
# ---------------------------------------------------------
st.markdown("#### 2. reBAP Settlement Spikes & Resulting Financial Cash Drawdowns")

fig2 = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
    subplot_titles=("Financial Cash Penalty (€) per Settlement Period", "reBAP vs. Day-Ahead Spot Price (€/MWh)")
)

fig2.add_trace(
    go.Bar(
        x=time_index, y=df_sim["cash_penalty_eur"],
        name="Cash Penalty (€)", marker_color="#f43f5e", opacity=0.85
    ),
    row=1, col=1
)

fig2.add_trace(
    go.Scatter(
        x=time_index, y=rebap_price,
        name="reBAP Price (€/MWh)", line=dict(color="#fbbf24", width=1.8)
    ),
    row=2, col=1
)
fig2.add_trace(
    go.Scatter(
        x=time_index, y=da_price,
        name="Day-Ahead Price (€/MWh)", line=dict(color="#94a3b8", width=1.5, dash="dot")
    ),
    row=2, col=1
)

fig2.update_layout(
    template="plotly_dark",
    plot_bgcolor="#0b0f19",
    paper_bgcolor="#0b0f19",
    height=500,
    margin=dict(l=20, r=20, t=50, b=20),
    legend=blue_legend_style,
    hovermode="x unified"
)
fig2.update_xaxes(gridcolor="#1e293b")
fig2.update_yaxes(gridcolor="#1e293b")

st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------
# Quantile Evaluation Metrics Row
# ---------------------------------------------------------
st.markdown("#### 3. Quantile Pinball Loss Evaluation")
l1, l2, l3 = st.columns(3)
l1.metric("P10 Pinball Loss", f"{p10_loss:.4f} GW")
l2.metric("P50 Pinball Loss (Median)", f"{p50_loss:.4f} GW")
l3.metric("P90 Pinball Loss", f"{p90_loss:.4f} GW")
