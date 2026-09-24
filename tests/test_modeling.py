import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from water_forecasting import features as F
from water_forecasting.modeling import (
    TrendNormalized,
    evaluate,
    fit_growth_trend,
    make_model,
    rolling_origin_folds,
    time_split,
)

FAMILIES = [
    "seasonal_naive_168",
    "ridge_fourier",
    "lightgbm",
    "lightgbm_trend",
    "ridge_interactions",
    "blend_ridge_lgbm",
]


@pytest.fixture(scope="module")
def data(raw):
    targets = F.hourly_targets("2025-03-22", "2025-05-10")
    feats = F.build_features(targets, raw["consumption"], raw["weather_fcst"], raw["calendar"], raw["weather_obs"])
    return F.attach_label(feats, raw["consumption"])


def test_time_split_never_overlaps(data):
    tr, ho = time_split(data, 7)
    assert tr.target_ts.max() < ho.target_ts.min()
    assert ho.target_ts.dt.normalize().nunique() == 7


def test_rolling_origin_is_expanding(data):
    folds = list(rolling_origin_folds(data, 3, 7))
    assert len(folds) == 3
    assert all(tr.target_ts.max() < te.target_ts.min() for tr, te in folds)
    assert len(folds[0][0]) < len(folds[-1][0])


@pytest.mark.parametrize("family", FAMILIES)
def test_every_family_fits_and_predicts(data, family, cfg):
    cols = F.model_columns()
    tr, ho = time_split(data, 7)
    params = {**cfg.model.lgbm_params, "n_estimators": 50}
    pred = make_model(family, 1.0, params).fit(tr[cols], tr.demand_m3h).predict(ho[cols])
    assert pred.shape == (len(ho),) and np.isfinite(pred).all()


@pytest.mark.parametrize("family", FAMILIES)
def test_optional_training_hooks_are_safe_for_every_family(data, family, cfg):
    """training.py calls feature_importance / annual_growth_pct when present - they must never raise."""
    cols = F.model_columns()
    tr, _ = time_split(data, 7)
    est = make_model(family, 1.0, {**cfg.model.lgbm_params, "n_estimators": 10}).fit(tr[cols], tr.demand_m3h)
    if hasattr(est, "feature_importance"):
        imp = est.feature_importance(cols)
        assert imp is None or len(imp) > 0
    if hasattr(est, "annual_growth_pct"):
        assert np.isfinite(est.annual_growth_pct)
    assert isinstance(est.get_params(), dict)


def test_every_configured_candidate_exists(cfg):
    for family in cfg.model.candidates:
        make_model(family, 1.0, cfg.model.lgbm_params)


def test_learned_models_beat_naive(data, cfg):
    cols = F.model_columns()
    tr, ho = time_split(data, 7)
    score = {
        f: evaluate(
            ho.demand_m3h,
            make_model(f, 1.0, cfg.model.lgbm_params).fit(tr[cols], tr.demand_m3h).predict(ho[cols]),
        )["mape"]
        for f in ["seasonal_naive_168", "ridge_fourier", "ridge_interactions"]
    }
    assert score["ridge_fourier"] < score["seasonal_naive_168"]
    assert score["ridge_interactions"] < score["seasonal_naive_168"]


def test_trend_normalisation_extrapolates_growth():
    """Demand growing 10 %/yr: the wrapper must forecast levels above anything seen in training."""
    t = np.arange(0, 730, 1 / 24)
    y = 100_000 * np.exp(np.log(1.10) * t / 365)
    X = pd.DataFrame({F.TIME_COL: t, "lag_168h": 100_000 * np.exp(np.log(1.10) * (t - 7) / 365)})
    train, future = X[t < 600], X[t >= 700]
    model = TrendNormalized(DummyRegressor(strategy="mean")).fit(train, y[t < 600])
    pred = model.predict(future)
    assert pred.min() > y[t < 600].max()  # beyond the training range
    assert model.annual_growth_pct == pytest.approx(10.0, abs=0.1)
    assert np.allclose(pred, y[t >= 700], rtol=0.01)


def test_seasonality_is_not_mistaken_for_growth():
    """Two years of flat demand with a +/-9 % annual cycle: the fitted growth must be ~0."""
    t = np.arange(0, 800, 1 / 24)
    y = 100_000 * (1 + 0.09 * np.sin(2 * np.pi * t / 365.25))
    _, slope = fit_growth_trend(t, y)
    assert abs(np.expm1(slope * 365)) < 0.005


def test_short_history_does_not_extrapolate_growth():
    t = np.arange(0, 400, 1 / 24)  # ~13 months: growth and season are not separable
    y = 100_000 * np.exp(np.log(1.10) * t / 365) * (1 + 0.09 * np.sin(2 * np.pi * t / 365.25))
    _, slope = fit_growth_trend(t, y)
    assert slope == 0.0


def test_raw_models_ignore_the_time_index(data, cfg):
    """The production raw models must not see t_days (they cannot extrapolate it)."""
    cols = F.model_columns()
    tr, _ = time_split(data, 7)
    m = make_model("lightgbm", lgbm_params={**cfg.model.lgbm_params, "n_estimators": 10}).fit(tr[cols], tr.demand_m3h)
    assert F.TIME_COL not in m.columns_


def test_metrics():
    m = evaluate(pd.Series([100.0, 200.0]), np.array([110.0, 180.0]))
    assert m["mape"] == pytest.approx(10.0)
    assert m["bias_pct"] == pytest.approx(-10 / 300 * 100)
