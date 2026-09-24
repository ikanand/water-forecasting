import pandas as pd
import pytest

from tests.conftest import EVENT_DATE, HOLIDAY_DATE, RAMADAN_DATES
from water_forecasting import features as F


def _build(raw, first, last, **kw):
    targets = F.hourly_targets(first, last)
    return F.build_features(targets, raw["consumption"], raw["weather_fcst"], raw["calendar"], raw["weather_obs"], **kw)


@pytest.fixture(scope="module")
def feats(raw):
    return _build(raw, "2025-04-01", "2025-05-12")


def test_one_row_per_hour(feats):
    assert not feats.duplicated(F.KEYS).any()
    assert len(feats) == 42 * 24
    assert list(feats.columns) == F.KEYS + F.model_columns()


def test_lag_values_are_correct(feats, raw):
    c = raw["consumption"].set_index("timestamp_local").demand_m3h
    row = feats[feats.target_ts == pd.Timestamp("2025-04-20 08:00")].iloc[0]
    assert row.lag_168h == pytest.approx(c[pd.Timestamp("2025-04-13 08:00")])
    assert row.lag_48h == pytest.approx(c[pd.Timestamp("2025-04-18 08:00")])


def test_no_leakage_tomorrow_is_fully_featured_without_its_actuals(raw):
    """Data ends 2025-05-10 (D-1). Forecasting 2025-05-12 must not need anything after 05-10."""
    f = _build(raw, "2025-05-12", "2025-05-12")
    assert f[F.model_columns()].notna().all().all()


def test_lags_below_min_are_rejected(raw):
    with pytest.raises(ValueError, match="leak"):
        _build(raw, "2025-05-01", "2025-05-01", min_lag_h=72)


def test_weather_uses_forecast_issued_day_before(feats, raw):
    f = raw["weather_fcst"]
    ts = pd.Timestamp("2025-04-20 15:00")
    issued_day = ts.normalize() - pd.Timedelta(days=1)
    expected = f[(f.timestamp_local == ts) & (f.forecast_issued_local.dt.normalize() == issued_day)]
    row = feats[feats.target_ts == ts].iloc[0]
    assert row.fc_temperature_c == pytest.approx(expected.forecast_temperature_c.iloc[0])


def test_temperature_persistence_features_only_use_the_past(feats, raw):
    ts = pd.Timestamp("2025-04-20 15:00")
    row = feats[feats.target_ts == ts].iloc[0]
    obs = raw["weather_obs"]
    d_minus_1 = obs[obs.timestamp_local.dt.normalize() == pd.Timestamp("2025-04-18")].temperature_c.mean()
    assert row.obs_tmean_d2 == pytest.approx(d_minus_1)  # observed D-1 for target day D+1
    f = raw["weather_fcst"]
    vintage = (f.timestamp_local.dt.normalize() == pd.Timestamp("2025-04-19")) & (
        f.forecast_issued_local.dt.normalize() == pd.Timestamp("2025-04-18")
    )
    prev = f[vintage].forecast_temperature_c.mean()  # the forecast for D, issued on D-1
    assert row.fc_tmean_prev_day == pytest.approx(prev)
    assert row.fc_cdd24_hour == pytest.approx(max(row.fc_temperature_c - F.COOLING_KNOT_C, 0))


def test_calendar_flags(feats):
    hol = feats[feats.target_ts.dt.normalize() == HOLIDAY_DATE]
    assert (hol.is_holiday == 1).all() and (hol.is_day_off == 1).all() and (hol.is_eid == 1).all()
    assert (feats[feats.target_ts.dt.normalize() != HOLIDAY_DATE].is_eid == 0).all()

    evt = feats[feats.target_ts.dt.normalize() == EVENT_DATE]
    assert (evt.event_structural == 1).all()
    assert (evt.event_operational == 0).all() and (evt.event_weather == 0).all()

    ram = feats[feats.target_ts.dt.normalize().isin(RAMADAN_DATES)]
    assert (ram.is_ramadan == 1).all() and (ram.is_school_break == 1).all()


def test_time_index_counts_days_since_epoch(feats):
    row = feats[feats.target_ts == pd.Timestamp("2025-04-01 12:00")].iloc[0]
    assert row[F.TIME_COL] == pytest.approx((pd.Timestamp("2025-04-01 12:00") - F.EPOCH) / pd.Timedelta(days=1))
