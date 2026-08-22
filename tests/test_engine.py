import numpy as np
from src.risk_metrics import pinball_loss, calculate_imbalance_pnl, compute_var_cvar


def test_pinball_loss_symmetry():
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([12.0, 18.0])
    loss_p50 = pinball_loss(y_true, y_pred, quantile=0.5)
    assert loss_p50 > 0.0
    assert np.isclose(loss_p50, 1.0)


def test_imbalance_pnl_calculation():
    actual = np.array([100.0, 50.0])
    forecast = np.array([90.0, 60.0])
    da_price = np.array([50.0, 50.0])
    rebap_price = np.array([150.0, 200.0])
    
    df = calculate_imbalance_pnl(actual, forecast, da_price, rebap_price)
    assert len(df) == 2
    assert "cash_penalty_eur" in df.columns


def test_cvar_greater_than_var():
    penalties = np.array([10, 20, 30, 40, 50, 100, 200, 500, 1000, 2500])
    stats = compute_var_cvar(penalties, confidence_level=0.90)
    assert stats["cvar"] >= stats["var"]
