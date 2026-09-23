import pandas as pd

from tests.conftest import ZONES
from water_forecasting import synthetic


def test_deterministic_day_by_day_equals_range(raw):
    """The daily replay must produce exactly what the backfill would have."""
    one = synthetic.generate_range("2025-04-15", "2025-04-15", ZONES, seed=7)
    ref = raw["consumption"]
    ref = ref[ref.reading_ts.dt.normalize() == pd.Timestamp("2025-04-15")].reset_index(drop=True)
    pd.testing.assert_frame_equal(one["consumption"].reset_index(drop=True), ref)


def test_shapes(raw):
    days = 71
    assert len(raw["consumption"]) == days * 24 * len(ZONES)
    assert len(raw["weather_fcst"]) == days * 48 * len(ZONES)
    assert (raw["consumption"].consumption_m3 > 0).all()


def test_forecast_issued_next_midnight_covers_two_days(raw):
    f = raw["weather_fcst"]
    assert f.issued_at.min() == pd.Timestamp("2025-03-02")
    assert f.lead_hours.between(1, 48).all()


def test_faults_break_the_data():
    ok = synthetic.generate_range("2025-04-15", "2025-04-15", ZONES, seed=7)
    no_fc = synthetic.generate_range("2025-04-15", "2025-04-15", ZONES, seed=7, fault="missing_forecast")
    outage = synthetic.generate_range("2025-04-15", "2025-04-15", ZONES, seed=7, fault="meter_outage")
    assert ZONES[0] not in set(no_fc["weather_fcst"].zone_id)
    assert len(outage["consumption"]) == len(ok["consumption"]) - 8
