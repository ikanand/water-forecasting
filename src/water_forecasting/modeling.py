"""Candidate model families + evaluation metrics. Pure Python (no MLflow, no Spark).

Candidates, from simplest to strongest:
* seasonal_naive_168 - "same hour last week". The bar every model must clear.
* ridge_fourier      - linear model on Fourier seasonality + weather + calendar. Transparent.
* lightgbm           - gradient boosting on all features. Usually the winner for hourly demand.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LINEAR_EXCLUDE = {"hour", "day_of_week", "month"}  # raw integers are meaningless to a linear model


class SeasonalNaive:
    """Predict the value `lag` hours ago (falls back to the 3-week same-hour mean)."""

    def __init__(self, lag_col: str = "lag_168h", fallback_col: str = "same_hour_mean_3w"):
        self.lag_col, self.fallback_col = lag_col, fallback_col

    def fit(self, X: pd.DataFrame, y=None):
        self.fallback_value_ = float(np.nanmean(y)) if y is not None else 0.0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pred = X[self.lag_col].fillna(X[self.fallback_col])
        return pred.fillna(getattr(self, "fallback_value_", 0.0)).to_numpy()

    def get_params(self, deep=True) -> dict[str, Any]:
        return {"lag_col": self.lag_col, "fallback_col": self.fallback_col}


class RidgeFourier:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X: pd.DataFrame, y):
        self.columns_ = [c for c in X.columns if c not in LINEAR_EXCLUDE]
        self.pipe_ = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=self.alpha))
        self.pipe_.fit(X[self.columns_], y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipe_.predict(X[self.columns_])

    def get_params(self, deep=True) -> dict[str, Any]:
        return {"alpha": self.alpha}


class LightGBMModel:
    def __init__(self, **params):
        self.params = params

    def fit(self, X: pd.DataFrame, y):
        import lightgbm as lgb

        self.model_ = lgb.LGBMRegressor(**self.params).fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X)

    def get_params(self, deep=True) -> dict[str, Any]:
        return dict(self.params)

    def feature_importance(self, columns: list[str]) -> pd.Series:
        return pd.Series(self.model_.booster_.feature_importance("gain"), index=columns).sort_values(ascending=False)


def make_model(name: str, ridge_alpha: float = 1.0, lgbm_params: dict | None = None):
    if name == "seasonal_naive_168":
        return SeasonalNaive("lag_168h")
    if name == "seasonal_naive_48":
        return SeasonalNaive("lag_48h")
    if name == "ridge_fourier":
        return RidgeFourier(alpha=ridge_alpha)
    if name == "lightgbm":
        return LightGBMModel(**(lgbm_params or {}))
    raise ValueError(f"Unknown model family {name!r}")


# ------------------------------------------------------------------ metrics ----
def evaluate(y_true, y_pred, hours: pd.Series | None = None) -> dict[str, float]:
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = p - y
    mask = y > 0
    out = {
        "mape": float(np.mean(np.abs(err[mask]) / y[mask]) * 100),
        "wape": float(np.sum(np.abs(err)) / np.sum(np.abs(y)) * 100),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias_pct": float(np.sum(err) / np.sum(y) * 100),
    }
    # accuracy where it hurts most operationally: the top-10% demand hours
    peak = y >= np.quantile(y, 0.9)
    out["peak_mape"] = float(np.mean(np.abs(err[peak]) / y[peak]) * 100)
    return out


def time_split(df: pd.DataFrame, holdout_days: int, ts_col: str = "target_ts"):
    """Last `holdout_days` full days = holdout. Never shuffle a time series."""
    cutoff = df[ts_col].max().normalize() - pd.Timedelta(days=holdout_days - 1)
    return df[df[ts_col] < cutoff], df[df[ts_col] >= cutoff]


def rolling_origin_folds(df: pd.DataFrame, n_folds: int, fold_days: int, ts_col: str = "target_ts"):
    """Expanding-window backtest: train on everything before each fold, test on the fold."""
    end = df[ts_col].max().normalize() + pd.Timedelta(days=1)
    for i in range(n_folds, 0, -1):
        test_start = end - pd.Timedelta(days=i * fold_days)
        test_end = test_start + pd.Timedelta(days=fold_days)
        yield df[df[ts_col] < test_start], df[(df[ts_col] >= test_start) & (df[ts_col] < test_end)]
