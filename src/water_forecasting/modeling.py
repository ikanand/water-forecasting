"""Candidate model families + evaluation metrics. Pure Python (no MLflow, no Spark).

Candidates (every weekly training run fits all of them and keeps the best on a fresh holdout):
* seasonal_naive_168 - "same hour last week". The bar every model must clear.
* ridge_fourier      - linear model on the raw target. Transparent reference.
* lightgbm           - gradient boosting on the raw target.
* lightgbm_trend     - LightGBM on a growth-normalised target (demand grows ~10 %/yr and trees
                       cannot extrapolate a level they have never seen).
* ridge_interactions - growth-normalised linear model with hour x (day off, Ramadan, cooling)
                       interactions - the EDA showed those effects reshape the day.
* blend_ridge_lgbm   - equal-weight average of the two growth-aware models.

Model selection evidence: notebooks/02_modelling.ipynb (12-fold rolling-origin backtest).
Every model receives features.model_columns() (the features plus the trend index t_days).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from water_forecasting.features import LEVEL_FEATURES, TIME_COL

LINEAR_EXCLUDE = {"hour", "day_of_week", "month", TIME_COL}  # raw integers mean nothing to a linear model


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


class RidgeInteractions:
    """Ridge on the features plus hour one-hot and hour x (day off, Ramadan, cooling degree) terms."""

    def __init__(self, alpha: float = 3.0):
        self.alpha = alpha

    def _design(self, X: pd.DataFrame) -> np.ndarray:
        hours = np.eye(24)[X["hour"].astype(int).to_numpy()]
        base = X.drop(columns=[c for c in LINEAR_EXCLUDE if c in X.columns])
        inter = np.hstack(
            [
                hours * X[["is_day_off"]].to_numpy(),
                hours * X[["is_ramadan"]].to_numpy(),
                hours * X[["fc_cdd24_hour"]].fillna(0).to_numpy(),
            ]
        )
        return np.hstack([base.to_numpy(float), hours, inter])

    def fit(self, X: pd.DataFrame, y):
        self.pipe_ = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=self.alpha))
        self.pipe_.fit(self._design(X), y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipe_.predict(self._design(X))

    def get_params(self, deep=True) -> dict[str, Any]:
        return {"alpha": self.alpha}


class LightGBMModel:
    def __init__(self, **params):
        self.params = params

    def fit(self, X: pd.DataFrame, y):
        import lightgbm as lgb

        self.columns_ = [c for c in X.columns if c != TIME_COL]
        self.model_ = lgb.LGBMRegressor(**self.params).fit(X[self.columns_], y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X[self.columns_])

    def get_params(self, deep=True) -> dict[str, Any]:
        return dict(self.params)

    def feature_importance(self, columns: list[str] | None = None) -> pd.Series:
        return pd.Series(self.model_.booster_.feature_importance("gain"), index=self.columns_).sort_values(
            ascending=False
        )


MIN_GROWTH_HISTORY_DAYS = 548  # ~18 months: below this, growth and the annual cycle cannot be separated


def fit_growth_trend(t_days, y) -> tuple[float, float]:
    """(intercept, slope) of log(daily mean demand) ~ a + b*t, with annual harmonics as nuisance terms so the
    seasonal swing is not mistaken for growth. With less than MIN_GROWTH_HISTORY_DAYS of history the slope is 0
    (level only). Evidence: notebooks/02_modelling.ipynb and the dev/qa short-history checks."""
    daily = pd.Series(np.asarray(y, float)).groupby(np.floor(np.asarray(t_days, float))).mean()
    t, ly = daily.index.to_numpy(float), np.log(daily.to_numpy())
    if t.max() - t.min() < MIN_GROWTH_HISTORY_DAYS:
        return float(np.mean(ly)), 0.0
    w = 2 * np.pi * t / 365.25
    design = np.column_stack([np.ones_like(t), t, np.sin(w), np.cos(w), np.sin(2 * w), np.cos(2 * w)])
    beta, *_ = np.linalg.lstsq(design, ly, rcond=None)
    return float(beta[0]), float(beta[1])


class TrendNormalized:
    """Growth-aware wrapper: fit a growth trend on the training history (fit_growth_trend), divide the target
    and every level-type feature by it, model the ratio, multiply back. The trend is extrapolated at
    prediction time, which is what raw tree models cannot do."""

    def __init__(self, model):
        self.model = model

    def _trend(self, t_days) -> np.ndarray:
        return np.exp(self.intercept_ + self.slope_ * np.asarray(t_days, float))

    def _normalise(self, X: pd.DataFrame):
        tr = self._trend(X[TIME_COL])
        Z = X.copy()
        for col in LEVEL_FEATURES:
            if col in Z:
                Z[col] = Z[col] / tr
        return Z, tr

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y, float)
        self.intercept_, self.slope_ = fit_growth_trend(X[TIME_COL].to_numpy(), y)
        Z, tr = self._normalise(X)
        self.model.fit(Z, y / tr)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Z, tr = self._normalise(X)
        return self.model.predict(Z) * tr

    @property
    def annual_growth_pct(self) -> float:
        return float(np.expm1(self.slope_ * 365) * 100)

    def get_params(self, deep=True) -> dict[str, Any]:
        return {"wrapped": type(self.model).__name__, **self.model.get_params()}

    def feature_importance(self, columns: list[str] | None = None) -> pd.Series | None:
        """The wrapped model's importances, or None if it has none (e.g. a linear model)."""
        inner = getattr(self.model, "feature_importance", None)
        return inner(columns) if inner is not None else None


class Blend:
    """Equal-weight average of several fitted models."""

    def __init__(self, models: list):
        self.models = models

    def fit(self, X: pd.DataFrame, y):
        for m in self.models:
            m.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.mean([m.predict(X) for m in self.models], axis=0)

    def get_params(self, deep=True) -> dict[str, Any]:
        return {f"member_{i}": type(getattr(m, "model", m)).__name__ for i, m in enumerate(self.models)}


def make_model(name: str, ridge_alpha: float = 1.0, lgbm_params: dict | None = None):
    lgbm_params = lgbm_params or {}
    if name == "seasonal_naive_168":
        return SeasonalNaive("lag_168h")
    if name == "seasonal_naive_48":
        return SeasonalNaive("lag_48h")
    if name == "ridge_fourier":
        return RidgeFourier(alpha=ridge_alpha)
    if name == "lightgbm":
        return LightGBMModel(**lgbm_params)
    if name == "lightgbm_trend":
        return TrendNormalized(LightGBMModel(**lgbm_params))
    if name == "ridge_interactions":
        return TrendNormalized(RidgeInteractions())
    if name == "blend_ridge_lgbm":
        return Blend([make_model("lightgbm_trend", lgbm_params=lgbm_params), make_model("ridge_interactions")])
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
