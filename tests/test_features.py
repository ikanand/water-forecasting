import numpy as np
import pandas as pd
import pytest

from tests.conftest import ZONES
from water_forecasting import features as F


@pytest.fixture(scope="module")
def feats(raw):
    targets = F.hourly_targets(ZONES, "2025-04-01", "2025-05-12")
    return F.build_features(targets, raw["consumption"], raw["weather_fcst"], raw["holidays"], raw["events"], ZONES)


def test_one_row_per_zone_hour(feats):
    assert not feats.duplicated(F.KEYS).any()
    assert len(feats) == 42 * 24 * len(ZONES)
    assert list(feats.columns) == F.KEYS + F.feature_columns(ZONES)


def test_lag_values_are_correct(feats, raw):
    c = raw["consumption"].set_index(["zone_id", "reading_ts"]).consumption_m3
    row = feats[(feats.zone_id == "central") & (feats.target_ts == pd.Timestamp("2025-04-20 08:00"))].iloc[0]
    assert row.lag_168h == pytest.approx(c[("central", pd.Timestamp("2025-04-13 08:00"))])
    assert row.lag_48h == pytest.approx(c[("central", pd.Timestamp("2025-04-18 08:00"))])


def test_no_leakage_tomorrow_is_fully_featured_without_its_actuals(raw):
    """Data ends 2025-05-10 (D-1). Forecasting 2025-05-12 must not need anything after 05-10."""
    targets = F.hourly_targets(ZONES, "2025-05-12", "2025-05-12")
    f = F.build_features(targets, raw["consumption"], raw["weather_fcst"], raw["holidays"], raw["events"], ZONES)
    assert f[F.feature_columns(ZONES)].notna().all().all()


def test_lags_below_min_are_rejected(raw):
    targets = F.hourly_targets(ZONES, "2025-05-01", "2025-05-01")
    with pytest.raises(ValueError, match="leak"):
        F.build_features(
            targets, raw["consumption"], raw["weather_fcst"], raw["holidays"], raw["events"], ZONES, min_lag_h=72
        )


def test_weather_uses_forecast_issued_day_before(feats, raw):
    f = raw["weather_fcst"]
    ts = pd.Timestamp("2025-04-20 15:00")
    expected = f[(f.zone_id == "south") & (f.target_ts == ts) & (f.issued_at == pd.Timestamp("2025-04-19"))]
    row = feats[(feats.zone_id == "south") & (feats.target_ts == ts)].iloc[0]
    assert row.fc_temperature_c == pytest.approx(expected.temperature_c.iloc[0])


def test_calendar_flags(feats):
    may1 = feats[feats.target_ts.dt.normalize() == pd.Timestamp("2025-05-01")]  # Labour Day
    assert (may1.is_holiday == 1).all() and (may1.is_day_off == 1).all()
    assert np.allclose(feats[["zone_north", "zone_central", "zone_south"]].sum(axis=1), 1)
