import numpy as np
import pandas as pd
import pytest

from tests.conftest import ZONES
from water_forecasting import features as F
from water_forecasting.modeling import evaluate, make_model, rolling_origin_folds, time_split


@pytest.fixture(scope="module")
def data(raw):
    targets = F.hourly_targets(ZONES, "2025-03-22", "2025-05-10")
    feats = F.build_features(targets, raw["consumption"], raw["weather_fcst"], raw["holidays"], raw["events"], ZONES)
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


@pytest.mark.parametrize("family", ["seasonal_naive_168", "ridge_fourier", "lightgbm"])
def test_every_family_fits_and_predicts(data, family, cfg):
    cols = F.feature_columns(ZONES)
    tr, ho = time_split(data, 7)
    params = {**cfg.model.lgbm_params, "n_estimators": 50}
    pred = make_model(family, 1.0, params).fit(tr[cols], tr.consumption_m3).predict(ho[cols])
    assert pred.shape == (len(ho),) and np.isfinite(pred).all()


def test_learned_models_beat_naive(data, cfg):
    cols = F.feature_columns(ZONES)
    tr, ho = time_split(data, 7)
    score = {
        f: evaluate(
            ho.consumption_m3,
            make_model(f, 1.0, cfg.model.lgbm_params).fit(tr[cols], tr.consumption_m3).predict(ho[cols]),
        )["mape"]
        for f in ["seasonal_naive_168", "ridge_fourier"]
    }
    assert score["ridge_fourier"] < score["seasonal_naive_168"]


def test_metrics():
    m = evaluate(pd.Series([100.0, 200.0]), np.array([110.0, 180.0]))
    assert m["mape"] == pytest.approx(10.0)
    assert m["bias_pct"] == pytest.approx(-10 / 300 * 100)
