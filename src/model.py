import lightgbm as lgb
import numpy as np
import pandas as pd


class QuantileEnergyForecaster:
    """Trains 3 LightGBM regressors targeting P10, P50, and P90 quantiles."""

    def __init__(self, quantiles=(0.10, 0.50, 0.90)):
        self.quantiles = quantiles
        self.models = {}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        for q in self.quantiles:
            reg = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=120,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42,
                verbose=-1
            )
            reg.fit(X, y)
            self.models[f"p{int(q*100)}"] = reg

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:
        preds = {}
        for name, model in self.models.items():
            preds[name] = model.predict(X)
        return pd.DataFrame(preds, index=X.index)
