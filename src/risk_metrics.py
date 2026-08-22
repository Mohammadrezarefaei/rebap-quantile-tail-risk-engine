import numpy as np
import pandas as pd


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Calculates the Pinball Loss for a given quantile."""
    error = y_true - y_pred
    return float(np.mean(np.maximum(quantile * error, (quantile - 1) * error)))


def calculate_imbalance_pnl(
    actual_mwh: np.ndarray,
    forecast_mwh: np.ndarray,
    da_price_eur: np.ndarray,
    rebap_price_eur: np.ndarray,
) -> pd.DataFrame:
    """Computes volumetric error and financial settlement exposure (EUR).

    Formula:
    Short position (forecast < actual): Extra energy sold at reBAP vs DA.
    Long position (forecast > actual): Energy bought back from TSO at reBAP.
    Financial Drawdown = (Actual - Forecast) * (reBAP - DA_Price)
    """
    imbalance_volume = actual_mwh - forecast_mwh  # MWh (Pos = Long System, Neg = Short)
    spread = rebap_price_eur - da_price_eur       # EUR/MWh
    
    # Financial penalty incurred due to imbalance settlement
    cash_penalty_eur = -1.0 * imbalance_volume * spread

    return pd.DataFrame({
        "actual_mwh": actual_mwh,
        "forecast_mwh": forecast_mwh,
        "residual_mwh": imbalance_volume,
        "da_price": da_price_eur,
        "rebap_price": rebap_price_eur,
        "cash_penalty_eur": cash_penalty_eur
    })


def compute_var_cvar(penalties: np.ndarray, confidence_level: float = 0.95) -> dict:
    """Calculates Value at Risk (VaR) and Conditional Value at Risk (CVaR / Expected Shortfall)."""
    clean_penalties = penalties[~np.isnan(penalties)]
    if len(clean_penalties) == 0:
        return {"var": 0.0, "cvar": 0.0}
    
    # Isolate loss distribution (positive penalty = loss)
    losses = clean_penalties[clean_penalties > 0]
    if len(losses) == 0:
        return {"var": 0.0, "cvar": 0.0}

    var_threshold = float(np.percentile(losses, confidence_level * 100))
    cvar_value = float(np.mean(losses[losses >= var_threshold]))

    return {
        "var": var_threshold,
        "cvar": cvar_value
    }
